using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettleTypeDisplayTypeRelationEntityConfiguration : IEntityTypeConfiguration<SettleTypeDisplayTypeRelationEntity>
    {
        public void Configure(EntityTypeBuilder<SettleTypeDisplayTypeRelationEntity> builder)
        {
            builder.ToTable("SettleTypeDisplayTypeRelation");

            builder.HasKey(x => new { x.SettleTypeID, x.DisplayTypeID });

            builder.Property(x => x.SettleTypeID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.DisplayTypeID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.IsFirstHalf)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Is5050)
                .HasColumnType("bit")
            ;

        }
    }
}
