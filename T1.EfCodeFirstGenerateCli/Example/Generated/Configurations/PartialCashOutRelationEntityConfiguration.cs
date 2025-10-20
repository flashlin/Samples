using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PartialCashOutRelationEntityConfiguration : IEntityTypeConfiguration<PartialCashOutRelationEntity>
    {
        public void Configure(EntityTypeBuilder<PartialCashOutRelationEntity> builder)
        {
            builder.ToTable("PartialCashOutRelation");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.OriginalTransId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime2")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.IsDeleted)
                .HasColumnType("bit")
                .IsRequired()
            ;

        }
    }
}
