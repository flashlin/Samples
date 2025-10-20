using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SportsBetExtraInfoEntityConfiguration : IEntityTypeConfiguration<SportsBetExtraInfoEntity>
    {
        public void Configure(EntityTypeBuilder<SportsBetExtraInfoEntity> builder)
        {
            builder.ToTable("SportsBetExtraInfo");

            builder.HasKey(x => x.TransId);

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.LeoEnumValue)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.MarketTypeId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SelectionId)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.SelectionTypeId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MatchStatTypeId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MarketDetails)
                .HasColumnType("nvarchar(4000)")
                .HasMaxLength(4000)
            ;

            builder.Property(x => x.SelectionDetails)
                .HasColumnType("nvarchar(4000)")
                .HasMaxLength(4000)
            ;

            builder.Property(x => x.BetTypeName)
                .HasColumnType("nvarchar(1000)")
                .HasMaxLength(1000)
            ;

            builder.Property(x => x.TraceBetId)
                .HasColumnType("bigint(19,0)")
            ;

        }
    }
}
